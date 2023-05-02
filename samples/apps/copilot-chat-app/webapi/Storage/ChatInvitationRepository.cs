// Copyright (c) Microsoft. All rights reserved.

using SemanticKernel.Service.Model;

namespace SemanticKernel.Service.Storage;

/// <summary>
/// A repository for chat invitations.
/// </summary>
public class ChatInvitationRepository : Repository<ChatInvitation>
{
    /// <summary>
    /// Initializes a new instance of the ChatInvitationRepository class.
    /// </summary>
    /// <param name="storageContext">The storage context.</param>
    public ChatInvitationRepository(IStorageContext<ChatInvitation> storageContext)
        : base(storageContext)
    {
    }

    /// <summary>
    /// Finds accepted chat invitations by user id.
    /// </summary>
    /// <param name="userId">The user id.</param>
    /// <returns>A list of chat sessions.</returns>
    public Task<IEnumerable<ChatInvitation>> FindAcceptedInvitationByUserIdAsync(string userId)
    {
        return base.StorageContext.QueryEntitiesAsync(e => e.UserId == userId && e.IsAccepted == true);
    }

    /// <summary>
    /// Finds chat invitations by user id and chat id.
    /// </summary>
    /// <param name="userId">The user id.</param>
    /// <param name="chatId">The chat id.</param>
    /// <returns></returns>
    public Task<IEnumerable<ChatInvitation>> FindInvitationByUserIdAndChatIdAsync(string userId, string chatId)
    {
        return base.StorageContext.QueryEntitiesAsync(e => e.UserId == userId && e.ChatId == chatId);
    }
}